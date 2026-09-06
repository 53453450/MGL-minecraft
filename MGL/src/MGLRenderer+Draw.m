/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+Draw.m
// Draw command encoding methods extracted from MGLRenderer.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_frame_activity.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_draw_issue.h"

/* === C helpers used by Draw and Batch methods === */
/* mglRendererProgramHasSampledResourceNamed is non-static so
 * MGLRenderer+Batch.m can also call it.  Declared in MGLRenderer+Draw_Private.h. */

bool mglRendererProgramHasSampledResourceNamed(Program *program, const char *name)
{
    if (!program || !name) {
        return false;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        for (int resType = 0; resType < MGL_MAX_SHADER_RESOURCES; resType++) {
            MGLShaderResourceList *resources = &program->shader_resources_list[stage][resType];
            for (GLuint i = 0; resources->list && i < resources->count; i++) {
                MGLShaderResource *res = &resources->list[i];
                if (res->name &&
                    strcmp(res->name, name) == 0 &&
                    mglRendererResourceLooksSamplerLike(res, resType)) {
                    return true;
                }
            }
        }
    }

    return false;
}

static MGLRenderer *mglRendererDrawTarget(GLMContext glm_ctx)
{
    return mglRendererForContext(glm_ctx);
}

void mglRendererDrawArrays(GLMContext glm_ctx,
                                 uint32_t mode, int32_t first, int32_t count)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlDrawArrays:glm_ctx mode:mode first:first count:count];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawElements(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlDrawElements:glm_ctx mode:mode count:count
                             type:type indices:indices];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawRangeElements(GLMContext glm_ctx, uint32_t mode,
    uint32_t start, uint32_t end, int32_t count, uint32_t type,
    const void *indices)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlDrawRangeElements:glm_ctx mode:mode start:start end:end
                                   count:count type:type indices:indices];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawArraysInstanced(GLMContext glm_ctx, uint32_t mode,
    int32_t first, int32_t count, int32_t instance_count)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlDrawArraysInstanced:glm_ctx mode:mode first:first
                                   count:count instancecount:instance_count];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawElementsInstanced(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlDrawElementsInstanced:glm_ctx mode:mode count:count
                                     type:type indices:indices
                            instancecount:instance_count];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawElementsBaseVertex(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices, int32_t base_vertex)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlDrawElementsBaseVertex:glm_ctx mode:mode count:count
                                      type:type indices:indices
                                basevertex:base_vertex];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawRangeElementsBaseVertex(GLMContext glm_ctx, uint32_t mode,
    uint32_t start, uint32_t end, int32_t count, uint32_t type,
    const void *indices, int32_t base_vertex)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlDrawRangeElementsBaseVertex:glm_ctx mode:mode
                                           start:start end:end count:count
                                            type:type indices:indices
                                      basevertex:base_vertex];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawElementsInstancedBaseVertex(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count, int32_t base_vertex)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlDrawElementsInstancedBaseVertex:glm_ctx mode:mode
                                               count:count type:type
                                             indices:indices
                                       instancecount:instance_count
                                          basevertex:base_vertex];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawArraysIndirect(GLMContext glm_ctx,
    uint32_t mode, const void *indirect)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlDrawArraysIndirect:glm_ctx mode:mode indirect:indirect];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawElementsIndirect(GLMContext glm_ctx,
    uint32_t mode, uint32_t type, const void *indirect)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlDrawElementsIndirect:glm_ctx mode:mode type:type
                                 indirect:indirect];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawArraysInstancedBaseInstance(GLMContext glm_ctx, uint32_t mode,
    int32_t first, int32_t count, int32_t instance_count,
    uint32_t base_instance)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlDrawArraysInstancedBaseInstance:glm_ctx mode:mode
                                               first:first count:count
                                       instancecount:instance_count
                                        baseinstance:base_instance];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawElementsInstancedBaseInstance(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count, uint32_t base_instance)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlDrawElementsInstancedBaseInstance:glm_ctx mode:mode
                                                 count:count type:type
                                               indices:indices
                                         instancecount:instance_count
                                          baseinstance:base_instance];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawElementsInstancedBaseVertexBaseInstance(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count, int32_t base_vertex, uint32_t base_instance)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlDrawElementsInstancedBaseVertexBaseInstance:glm_ctx
                                                           mode:mode count:count
                                                           type:type
                                                        indices:indices
                                                  instancecount:instance_count
                                                     basevertex:base_vertex
                                                   baseinstance:base_instance];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererMultiDrawArrays(GLMContext glm_ctx, uint32_t mode,
    const int32_t *firsts, const int32_t *counts, int32_t draw_count)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlMultiDrawArrays:glm_ctx mode:mode
                               first:(const GLint *)firsts
                               count:(const GLsizei *)counts
                           drawcount:draw_count];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererMultiDrawElements(GLMContext glm_ctx, uint32_t mode,
    const int32_t *counts, uint32_t type, const void *const *indices,
    int32_t draw_count)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlMultiDrawElements:glm_ctx mode:mode
                                count:(const GLsizei *)counts type:type
                              indices:indices drawcount:draw_count];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererMultiDrawElementsBaseVertex(GLMContext glm_ctx, uint32_t mode,
    const int32_t *counts, uint32_t type, const void *const *indices,
    int32_t draw_count, const int32_t *base_vertices)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlMultiDrawElementsBaseVertex:glm_ctx mode:mode
                                          count:(const GLsizei *)counts
                                           type:type indices:indices
                                      drawcount:draw_count
                                     basevertex:(const GLint *)base_vertices];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererMultiDrawArraysIndirect(GLMContext glm_ctx, uint32_t mode,
    const void *indirect, int32_t draw_count, int32_t stride)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlMultiDrawArraysIndirect:glm_ctx mode:mode
                                    indirect:indirect drawcount:draw_count
                                       stride:stride];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererMultiDrawElementsIndirect(GLMContext glm_ctx, uint32_t mode, uint32_t type,
    const void *indirect, int32_t draw_count, int32_t stride)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    MGLRenderer *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    @autoreleasepool {
        [renderer mtlMultiDrawElementsIndirect:glm_ctx mode:mode type:type
                                      indirect:indirect drawcount:draw_count
                                         stride:stride];
    }
    mglRendererBackendEnd(&_backend_lease);
}


@implementation MGLRenderer (Draw)

-(void) mtlDrawArrays: (GLMContext) ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count
{
    self->_lastDrawPrimitiveMode = mode;

    METAL_LOCK();
    if ([self runEmulatedMSSampleDrawLoopIfNeeded:ctx drawOnce:^{
            [self mtlDrawArraysLocked:ctx mode:mode first:first count:count];
        }]) {
        METAL_UNLOCK();
        return;
    }
    [self mtlDrawArraysLocked:ctx mode:mode first:first count:count];
    [self broadcastEmulatedMSSamplePlanesAfterDrawIfNeeded:ctx];
    METAL_UNLOCK();
}

-(void) mtlDrawArraysLocked: (GLMContext) ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count
{
    mglIssueDrawArrays(ctx, (__bridge void *)self, mode, first, count, 1, 0u, "drawArrays");
}

-(void) mtlDrawElements: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices
{
    self->_lastDrawPrimitiveMode = mode;

    METAL_LOCK();
    if ([self runEmulatedMSSampleDrawLoopIfNeeded:glm_ctx drawOnce:^{
            [self mtlDrawElementsLocked:glm_ctx mode:mode count:count type:type
                                indices:indices];
        }]) {
        METAL_UNLOCK();
        return;
    }
    [self mtlDrawElementsLocked:glm_ctx mode:mode count:count type:type indices:indices];
    [self broadcastEmulatedMSSamplePlanesAfterDrawIfNeeded:glm_ctx];
    METAL_UNLOCK();
}

-(void) mtlDrawElementsLocked: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices
{
    mglIssueDrawElements(glm_ctx, (__bridge void *)self, mode, count, type, indices, 1, 0, 0u, "drawElements");
}

-(void) mtlDrawRangeElements: (GLMContext) glm_ctx mode:(GLenum) mode start:(GLuint) start end:(GLuint) end count: (GLsizei) count type: (GLenum) type indices:(const void *)indices
{
    (void)start;
    (void)end;
    mglIssueDrawElements(glm_ctx, (__bridge void *)self, mode, count, type, indices, 1, 0, 0u, "drawRangeElements");
}

-(void) mtlDrawArraysInstanced: (GLMContext) glm_ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count instancecount:(GLsizei) instancecount
{
    mglIssueDrawArrays(glm_ctx, (__bridge void *)self, mode, first, count, instancecount, 0u, "drawArraysInstanced");
}

-(void) mtlDrawElementsInstanced: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices instancecount:(GLsizei) instancecount
{
    mglIssueDrawElements(glm_ctx, (__bridge void *)self, mode, count, type, indices, instancecount, 0, 0u, "drawElementsInstanced");
}

-(void) mtlDrawElementsBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices basevertex:(GLint) basevertex
{
    mglIssueDrawElements(glm_ctx, (__bridge void *)self, mode, count, type, indices, 1, basevertex, 0u, "drawElementsBaseVertex");
}

-(void) mtlDrawRangeElementsBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode start: (GLuint) start end: (GLuint) end count:(GLsizei) count type: (GLenum) type indices:(const void *)indices basevertex:(GLint) basevertex
{
    (void)start;
    (void)end;
    mglIssueDrawElements(glm_ctx, (__bridge void *)self, mode, count, type, indices, 1, basevertex, 0u, "drawRangeElementsBaseVertex");
}

-(void) mtlDrawElementsInstancedBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode count:(GLsizei) count type: (GLenum) type indices:(const void *)indices instancecount:(GLsizei) instancecount basevertex:(GLint) basevertex
{
    mglIssueDrawElements(glm_ctx, (__bridge void *)self, mode, count, type, indices, instancecount, basevertex, 0u, "drawElementsInstancedBaseVertex");
}

-(void) mtlDrawArraysIndirect: (GLMContext) glm_ctx mode:(GLenum) mode indirect: (const void *) indirect
{
    mglIssueDrawArraysIndirect(glm_ctx, (__bridge void *)self, mode, indirect,
                               "drawArraysIndirect");
}

-(void) mtlDrawElementsIndirect: (GLMContext) glm_ctx mode:(GLenum) mode type:(GLenum) type indirect: (const void *) indirect
{
    mglIssueDrawElementsIndirect(glm_ctx, (__bridge void *)self, mode, type,
                                 indirect, "drawElementsIndirect");
}

-(void) mtlDrawArraysInstancedBaseInstance: (GLMContext) glm_ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count instancecount:(GLsizei) instancecount baseinstance:(GLuint) baseinstance
{
    mglIssueDrawArrays(glm_ctx, (__bridge void *)self, mode, first, count, instancecount, baseinstance, "drawArraysInstancedBaseInstance");
}

-(void) mtlDrawElementsInstancedBaseInstance: (GLMContext) glm_ctx mode:(GLenum) mode  count: (GLsizei) count type:(GLenum) type indices:(const void *)indices instancecount:(GLsizei) instancecount baseinstance:(GLuint) baseinstance
{
    mglIssueDrawElements(glm_ctx, (__bridge void *)self, mode, count, type, indices, instancecount, 0, baseinstance, "drawElementsInstancedBaseInstance");
}

-(void) mtlDrawElementsInstancedBaseVertexBaseInstance: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type:(GLenum) type indices:(const void *)indices
                                                        instancecount:(GLsizei) instancecount basevertex:(GLint) basevertex baseinstance:(GLuint) baseinstance
{
    mglIssueDrawElements(glm_ctx, (__bridge void *)self, mode, count, type, indices, instancecount, basevertex, baseinstance, "drawElementsInstancedBaseVertexBaseInstance");
}

-(void) mtlMultiDrawArrays: (GLMContext)glm_ctx mode:(GLenum) mode first:(const GLint *)first count:(const GLsizei *)count drawcount:(GLsizei) drawcount
{
    mglIssueMultiDrawArrays(glm_ctx, (__bridge void *)self, mode, first, count,
                            drawcount, "multiDrawArrays");
}

-(void) mtlMultiDrawElements: (GLMContext)glm_ctx mode:(GLenum) mode count:(const GLsizei *)count type:(GLenum)type indices:(const void *const*)indices drawcount:(GLsizei) drawcount
{
    mglIssueMultiDrawElements(glm_ctx, (__bridge void *)self, mode, count, type,
                              indices, drawcount, NULL, "multiDrawElements");
}

-(void) mtlMultiDrawElementsBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode count: (const GLsizei *) count type: (GLenum) type indices:(const void *const *)indices drawcount:(GLsizei) drawcount basevertex:(const GLint *) basevertex
{
    mglIssueMultiDrawElements(glm_ctx, (__bridge void *)self, mode, count, type,
                              indices, drawcount, basevertex,
                              "multiDrawElementsBaseVertex");
}

-(void) mtlMultiDrawArraysIndirect: (GLMContext)glm_ctx mode:(GLenum) mode indirect:(const void *)indirect drawcount:(GLsizei) drawcount stride:(GLsizei)stride
{
    mglIssueMultiDrawArraysIndirect(glm_ctx, (__bridge void *)self, mode,
                                    indirect, drawcount, stride,
                                    "multiDrawArraysIndirect");
}

-(void) mtlMultiDrawElementsIndirect: (GLMContext)glm_ctx mode:(GLenum) mode type:(GLenum)type indirect:(const void *)indirect drawcount:(GLsizei) drawcount stride:(GLsizei)stride
{
    mglIssueMultiDrawElementsIndirect(glm_ctx, (__bridge void *)self, mode, type,
                                      indirect, drawcount, stride,
                                      "multiDrawElementsIndirect");
}

@end
