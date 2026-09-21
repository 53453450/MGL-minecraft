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

#include "glm_context.h"
#include "mgl_renderer_backend.h"  /* MGLRendererBackendLease, lease begin/end */
#include "mgl_frame_activity.h"
#include "mgl_env_flag.h"
#include "mgl_draw_issue.h"
#include "mgl_sampler_compat.h"    /* mglRendererResourceLooksSamplerLike */

#include <stddef.h>
#include <string.h>

/* @autoreleasepool is objc_autoreleasePoolPush/Pop underneath; these entry
 * points keep the per-draw pool without making this file Objective-C. */
extern void *objc_autoreleasePoolPush(void);
extern void objc_autoreleasePoolPop(void *);

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

static void *mglRendererDrawTarget(GLMContext glm_ctx)
{
    return glm_ctx ? glm_ctx->platform_renderer_shell : NULL;
}

/* The Objective-C side has the same two-line inline in MGLRenderer_Private.h. */
static inline int mglRendererEnterBackendLease(GLMContext context,
                                               MGLRendererBackendLease *lease)
{
    return mglRendererBackendBeginContext(context, lease);
}

void mglRendererDrawArrays(GLMContext glm_ctx,
                                 uint32_t mode, int32_t first, int32_t count)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        mglDrawHostGuardIssueArrays(renderer, glm_ctx, mode, first, count, 1, 0u, "drawArrays", /*with_ms=*/1);
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawElements(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        mglDrawHostGuardIssueElements(renderer, glm_ctx, mode, count, type, indices, 1, 0, 0u, "drawElements", /*with_ms=*/1);
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawRangeElements(GLMContext glm_ctx, uint32_t mode,
    uint32_t start, uint32_t end, int32_t count, uint32_t type,
    const void *indices)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        (void)start; (void)end; mglIssueDrawElements(glm_ctx, renderer, mode, count, type, indices, 1, 0, 0u, "drawRangeElements");
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawArraysInstanced(GLMContext glm_ctx, uint32_t mode,
    int32_t first, int32_t count, int32_t instance_count)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        mglIssueDrawArrays(glm_ctx, renderer, mode, first, count, instance_count, 0u, "drawArraysInstanced");
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawElementsInstanced(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        mglIssueDrawElements(glm_ctx, renderer, mode, count, type, indices, instance_count, 0, 0u, "drawElementsInstanced");
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawElementsBaseVertex(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices, int32_t base_vertex)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        mglIssueDrawElements(glm_ctx, renderer, mode, count, type, indices, 1, base_vertex, 0u, "drawElementsBaseVertex");
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawRangeElementsBaseVertex(GLMContext glm_ctx, uint32_t mode,
    uint32_t start, uint32_t end, int32_t count, uint32_t type,
    const void *indices, int32_t base_vertex)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        (void)start; (void)end; mglIssueDrawElements(glm_ctx, renderer, mode, count, type, indices, 1, base_vertex, 0u, "drawRangeElementsBaseVertex");
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawElementsInstancedBaseVertex(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count, int32_t base_vertex)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        mglIssueDrawElements(glm_ctx, renderer, mode, count, type, indices, instance_count, base_vertex, 0u, "drawElementsInstancedBaseVertex");
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawArraysIndirect(GLMContext glm_ctx,
    uint32_t mode, const void *indirect)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        mglIssueDrawArraysIndirect(glm_ctx, renderer, mode, indirect, "drawArraysIndirect");
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawElementsIndirect(GLMContext glm_ctx,
    uint32_t mode, uint32_t type, const void *indirect)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        mglIssueDrawElementsIndirect(glm_ctx, renderer, mode, type, indirect, "drawElementsIndirect");
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawArraysInstancedBaseInstance(GLMContext glm_ctx, uint32_t mode,
    int32_t first, int32_t count, int32_t instance_count,
    uint32_t base_instance)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        mglIssueDrawArrays(glm_ctx, renderer, mode, first, count, instance_count, base_instance, "drawArraysInstancedBaseInstance");
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawElementsInstancedBaseInstance(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count, uint32_t base_instance)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        mglIssueDrawElements(glm_ctx, renderer, mode, count, type, indices, instance_count, 0, base_instance, "drawElementsInstancedBaseInstance");
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDrawElementsInstancedBaseVertexBaseInstance(GLMContext glm_ctx, uint32_t mode,
    int32_t count, uint32_t type, const void *indices,
    int32_t instance_count, int32_t base_vertex, uint32_t base_instance)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        mglIssueDrawElements(glm_ctx, renderer, mode, count, type, indices, instance_count, base_vertex, base_instance, "drawElementsInstancedBaseVertexBaseInstance");
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererMultiDrawArrays(GLMContext glm_ctx, uint32_t mode,
    const int32_t *firsts, const int32_t *counts, int32_t draw_count)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        mglIssueMultiDrawArrays(glm_ctx, renderer, mode, firsts, counts, draw_count, "multiDrawArrays");
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererMultiDrawElements(GLMContext glm_ctx, uint32_t mode,
    const int32_t *counts, uint32_t type, const void *const *indices,
    int32_t draw_count)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        mglIssueMultiDrawElements(glm_ctx, renderer, mode, counts, type, indices, draw_count, NULL, "multiDrawElements");
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererMultiDrawElementsBaseVertex(GLMContext glm_ctx, uint32_t mode,
    const int32_t *counts, uint32_t type, const void *const *indices,
    int32_t draw_count, const int32_t *base_vertices)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        mglIssueMultiDrawElements(glm_ctx, renderer, mode, counts, type, indices, draw_count, base_vertices, "multiDrawElementsBaseVertex");
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererMultiDrawArraysIndirect(GLMContext glm_ctx, uint32_t mode,
    const void *indirect, int32_t draw_count, int32_t stride)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        mglIssueMultiDrawArraysIndirect(glm_ctx, renderer, mode, indirect, draw_count, stride, "multiDrawArraysIndirect");
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererMultiDrawElementsIndirect(GLMContext glm_ctx, uint32_t mode, uint32_t type,
    const void *indirect, int32_t draw_count, int32_t stride)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;

    void *renderer = mglRendererDrawTarget(glm_ctx);
    if (!renderer) {
        mglRendererBackendEnd(&_backend_lease);
        return;
    }
    void *mgl_pool = objc_autoreleasePoolPush();
        mglIssueMultiDrawElementsIndirect(glm_ctx, renderer, mode, type, indirect, draw_count, stride, "multiDrawElementsIndirect");
    objc_autoreleasePoolPop(mgl_pool);
    mglRendererBackendEnd(&_backend_lease);
}
