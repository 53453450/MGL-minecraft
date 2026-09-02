/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * ObjC draw dispatch bridge — thin entry from C++ facade to MGLRenderer (Draw).
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"

static MGLRenderer *mglRendererDrawTarget(GLMContext glm_ctx)
{
    return mglRendererForContext(glm_ctx);
}

void mglRendererObjCDrawArrays(GLMContext glm_ctx,
                                 uint32_t mode, int32_t first, int32_t count)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlDrawArrays:glm_ctx mode:mode first:first count:count];
    }
}

void mglRendererObjCDrawElements(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlDrawElements:glm_ctx mode:mode count:count
                             type:type indices:indices];
    }
}

void mglRendererObjCDrawRangeElements(GLMContext glm_ctx, uint32_t mode,
    uint32_t start, uint32_t end, int32_t count, uint32_t type,
    const void *indices)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlDrawRangeElements:glm_ctx mode:mode start:start end:end
                                   count:count type:type indices:indices];
    }
}

void mglRendererObjCDrawArraysInstanced(GLMContext glm_ctx, uint32_t mode,
    int32_t first, int32_t count, int32_t instance_count)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlDrawArraysInstanced:glm_ctx mode:mode first:first
                                   count:count instancecount:instance_count];
    }
}

void mglRendererObjCDrawElementsInstanced(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlDrawElementsInstanced:glm_ctx mode:mode count:count
                                     type:type indices:indices
                            instancecount:instance_count];
    }
}

void mglRendererObjCDrawElementsBaseVertex(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices, int32_t base_vertex)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlDrawElementsBaseVertex:glm_ctx mode:mode count:count
                                      type:type indices:indices
                                basevertex:base_vertex];
    }
}

void mglRendererObjCDrawRangeElementsBaseVertex(GLMContext glm_ctx, uint32_t mode,
    uint32_t start, uint32_t end, int32_t count, uint32_t type,
    const void *indices, int32_t base_vertex)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlDrawRangeElementsBaseVertex:glm_ctx mode:mode
                                           start:start end:end count:count
                                            type:type indices:indices
                                      basevertex:base_vertex];
    }
}

void mglRendererObjCDrawElementsInstancedBaseVertex(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count, int32_t base_vertex)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlDrawElementsInstancedBaseVertex:glm_ctx mode:mode
                                               count:count type:type
                                             indices:indices
                                       instancecount:instance_count
                                          basevertex:base_vertex];
    }
}

void mglRendererObjCDrawArraysIndirect(GLMContext glm_ctx,
    uint32_t mode, const void *indirect)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlDrawArraysIndirect:glm_ctx mode:mode indirect:indirect];
    }
}

void mglRendererObjCDrawElementsIndirect(GLMContext glm_ctx,
    uint32_t mode, uint32_t type, const void *indirect)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlDrawElementsIndirect:glm_ctx mode:mode type:type
                                 indirect:indirect];
    }
}

void mglRendererObjCDrawArraysInstancedBaseInstance(GLMContext glm_ctx, uint32_t mode,
    int32_t first, int32_t count, int32_t instance_count,
    uint32_t base_instance)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlDrawArraysInstancedBaseInstance:glm_ctx mode:mode
                                               first:first count:count
                                       instancecount:instance_count
                                        baseinstance:base_instance];
    }
}

void mglRendererObjCDrawElementsInstancedBaseInstance(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count, uint32_t base_instance)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlDrawElementsInstancedBaseInstance:glm_ctx mode:mode
                                                 count:count type:type
                                               indices:indices
                                         instancecount:instance_count
                                          baseinstance:base_instance];
    }
}

void mglRendererObjCDrawElementsInstancedBaseVertexBaseInstance(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count, int32_t base_vertex, uint32_t base_instance)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlDrawElementsInstancedBaseVertexBaseInstance:glm_ctx
                                                           mode:mode count:count
                                                           type:type
                                                        indices:indices
                                                  instancecount:instance_count
                                                     basevertex:base_vertex
                                                   baseinstance:base_instance];
    }
}

void mglRendererObjCMultiDrawArrays(GLMContext glm_ctx, uint32_t mode,
    const int32_t *firsts, const int32_t *counts, int32_t draw_count)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlMultiDrawArrays:glm_ctx mode:mode
                               first:(const GLint *)firsts
                               count:(const GLsizei *)counts
                           drawcount:draw_count];
    }
}

void mglRendererObjCMultiDrawElements(GLMContext glm_ctx, uint32_t mode,
    const int32_t *counts, uint32_t type, const void *const *indices,
    int32_t draw_count)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlMultiDrawElements:glm_ctx mode:mode
                                count:(const GLsizei *)counts type:type
                              indices:indices drawcount:draw_count];
    }
}

void mglRendererObjCMultiDrawElementsBaseVertex(GLMContext glm_ctx, uint32_t mode,
    const int32_t *counts, uint32_t type, const void *const *indices,
    int32_t draw_count, const int32_t *base_vertices)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlMultiDrawElementsBaseVertex:glm_ctx mode:mode
                                          count:(const GLsizei *)counts
                                           type:type indices:indices
                                      drawcount:draw_count
                                     basevertex:(const GLint *)base_vertices];
    }
}

void mglRendererObjCMultiDrawArraysIndirect(GLMContext glm_ctx, uint32_t mode,
    const void *indirect, int32_t draw_count, int32_t stride)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlMultiDrawArraysIndirect:glm_ctx mode:mode
                                    indirect:indirect drawcount:draw_count
                                       stride:stride];
    }
}

void mglRendererObjCMultiDrawElementsIndirect(GLMContext glm_ctx, uint32_t mode, uint32_t type,
    const void *indirect, int32_t draw_count, int32_t stride)
{
    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) return;
    @autoreleasepool {
        [renderer mtlMultiDrawElementsIndirect:glm_ctx mode:mode type:type
                                      indirect:indirect drawcount:draw_count
                                         stride:stride];
    }
}
