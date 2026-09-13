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
        mglDrawHostGuardIssueArrays((__bridge void *)renderer, glm_ctx, mode, first, count, 1, 0u, "drawArrays", /*with_ms=*/1);
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
        mglDrawHostGuardIssueElements((__bridge void *)renderer, glm_ctx, mode, count, type, indices, 1, 0, 0u, "drawElements", /*with_ms=*/1);
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
        (void)start; (void)end; mglIssueDrawElements(glm_ctx, (__bridge void *)renderer, mode, count, type, indices, 1, 0, 0u, "drawRangeElements");
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
        mglIssueDrawArrays(glm_ctx, (__bridge void *)renderer, mode, first, count, instance_count, 0u, "drawArraysInstanced");
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
        mglIssueDrawElements(glm_ctx, (__bridge void *)renderer, mode, count, type, indices, instance_count, 0, 0u, "drawElementsInstanced");
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
        mglIssueDrawElements(glm_ctx, (__bridge void *)renderer, mode, count, type, indices, 1, base_vertex, 0u, "drawElementsBaseVertex");
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
        (void)start; (void)end; mglIssueDrawElements(glm_ctx, (__bridge void *)renderer, mode, count, type, indices, 1, base_vertex, 0u, "drawRangeElementsBaseVertex");
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
        mglIssueDrawElements(glm_ctx, (__bridge void *)renderer, mode, count, type, indices, instance_count, base_vertex, 0u, "drawElementsInstancedBaseVertex");
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
        mglIssueDrawArraysIndirect(glm_ctx, (__bridge void *)renderer, mode, indirect, "drawArraysIndirect");
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
        mglIssueDrawElementsIndirect(glm_ctx, (__bridge void *)renderer, mode, type, indirect, "drawElementsIndirect");
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
        mglIssueDrawArrays(glm_ctx, (__bridge void *)renderer, mode, first, count, instance_count, base_instance, "drawArraysInstancedBaseInstance");
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
        mglIssueDrawElements(glm_ctx, (__bridge void *)renderer, mode, count, type, indices, instance_count, 0, base_instance, "drawElementsInstancedBaseInstance");
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
        mglIssueDrawElements(glm_ctx, (__bridge void *)renderer, mode, count, type, indices, instance_count, base_vertex, base_instance, "drawElementsInstancedBaseVertexBaseInstance");
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
        mglIssueMultiDrawArrays(glm_ctx, (__bridge void *)renderer, mode, firsts, counts, draw_count, "multiDrawArrays");
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
        mglIssueMultiDrawElements(glm_ctx, (__bridge void *)renderer, mode, counts, type, indices, draw_count, NULL, "multiDrawElements");
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
        mglIssueMultiDrawElements(glm_ctx, (__bridge void *)renderer, mode, counts, type, indices, draw_count, base_vertices, "multiDrawElementsBaseVertex");
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
        mglIssueMultiDrawArraysIndirect(glm_ctx, (__bridge void *)renderer, mode, indirect, draw_count, stride, "multiDrawArraysIndirect");
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
        mglIssueMultiDrawElementsIndirect(glm_ctx, (__bridge void *)renderer, mode, type, indirect, draw_count, stride, "multiDrawElementsIndirect");
    }
    mglRendererBackendEnd(&_backend_lease);
}


@implementation MGLRenderer (Draw)

@end
