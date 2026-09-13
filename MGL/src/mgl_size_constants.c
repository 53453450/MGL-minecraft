/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_size_constants.c — -[MGLRenderer bindBufferSizeConstantsForRenderEncoder]
 * moved out of MGLRenderer+RenderPass.m (P0-1, log 104).
 *
 * Translation notes: `ctx` -> areas.ctx, MGL_STATE(ctx) -> the dual-proxy twin
 * below, `_backend` -> areas.backend, `_device` -> the device of that backend,
 * `_renderPassManager.state->currentRenderEncoderOwner` -> areas.command,
 * `[self …]` -> the C entries of mgl_binding_state_ops.h, and the file-local
 * mglRenderPassCreateBufferWithBytes of +RenderPass.m (which ignored its device
 * argument) -> a direct mglRenderCreateBufferWithBytes call.
 */

#include <string.h>

#include <CoreFoundation/CoreFoundation.h>

#include "mgl_size_constants.h"
#include "glm_context.h"
#include "mgl_renderer_ports.h"        /* state areas */
#include "mgl_renderer_backend.h"      /* size-constant cache + device */
#include "mgl_render.h"                /* the mglRender* facade */
#include "mgl_buffer_slots.h"          /* MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX */
#include "mgl_binding_state_ops.h"     /* record last bound buffer */
#include "mgl_frame_activity.h"        /* MGL_PERF_INC */
#include "mgl_types_buffer.h"          /* mglBufferMapVisibleSize */

/* MGL_STATE() from MGLRenderer_Private.h, in C. */
static GLMState *mglSizeConstantsState(const MGLRendererStateAreas *areas)
{
    if (areas->core && areas->core->activeState) {
        return areas->core->activeState;
    }
    return areas->ctx ? areas->ctx->active_state : NULL;
}

/* Releases a +1 the facade handed back from a create: mglRenderCreateBuffer*
 * does not bump the MGLMetalKind counters, so this stays a plain release (which
 * is what ARC did for the same reference). */
static void mglSizeConstantsReleaseCreated(void *buffer, void **slot)
{
    if (buffer) {
        CFRelease((CFTypeRef)buffer);
    }
    if (slot) {
        *slot = NULL;
    }
}

/* The size-constant buffer of one stage: the backend's cached one, else a fresh
 * shared-storage buffer stored in that cache.  Borrowed; NULL when the stage
 * needs nothing or the materialization failed. */
static void *mglSizeConstantsBufferForStage(const MGLRendererStateAreas *areas,
                                            uint32_t backend_kind,
                                            const uint32_t *size_constants,
                                            size_t constants_bytes)
{
    void *buffer = mglRendererBackendGetSizeConstantsBuffer(
        areas->backend, backend_kind, size_constants, 31u);
    if (buffer) {
        return buffer;
    }
    void *created = NULL;
    if (mglRenderCreateBufferWithBytes(size_constants, constants_bytes,
                                       MGLResourceStorageModeShared, NULL,
                                       &created) != 0 || !created) {
        return NULL;
    }
    if (mglRendererBackendSetSizeConstantsBuffer(
            areas->backend, backend_kind, size_constants, 31u, created) != 0) {
        mglSizeConstantsReleaseCreated(created, NULL);
        return NULL;
    }
    /* The backend cache holds its own reference now; drop ours the way ARC
     * dropped the strong local at the end of the method. */
    mglSizeConstantsReleaseCreated(created, NULL);
    return created;
}

/* The constants for one map list: the visible size of every mapped buffer, at
 * its Metal slot.  MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX is the array-size buffer
 * itself and never carries a size constant. */
static void mglSizeConstantsFillFromMapList(const BufferMapList *list,
                                            uint32_t *size_constants)
{
    memset(size_constants, 0, 31 * sizeof(uint32_t));
    for (int i = 0; i < list->count; i++) {
        const BufferMap *map = &list->buffers[i];
        if (!map->buf) {
            continue;
        }
        const size_t metalSlot = map->has_metal_binding
            ? (size_t)map->metal_binding_index
            : (size_t)map->buffer_base_index;
        if (metalSlot >= 31 || metalSlot == MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX) {
            continue;
        }
        size_constants[metalSlot] = (uint32_t)mglBufferMapVisibleSize(map);
    }
}

bool mglRendererBindBufferSizeConstantsForRenderEncoder(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    GLMState *state = mglSizeConstantsState(&areas);
    void *renderEncoderOwner =
        areas.command ? areas.command->currentRenderEncoderOwner : NULL;

    if (mglRenderEncoderOwnerHasCurrent(renderEncoderOwner) != 1) {
        return true;
    }

    Program *vertexProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    if (vertexProgram && vertexProgram->modules[_VERTEX_SHADER].needs_runtime_array_size_buffer)
    {
        uint32_t sizeConstants[31];
        mglSizeConstantsFillFromMapList(&state->vertex_buffer_map_list, sizeConstants);

        void *vertexSizeBuffer = mglSizeConstantsBufferForStage(
            &areas, MGL_RENDERER_BACKEND_SIZE_CONSTANTS_VERTEX,
            sizeConstants, sizeof(sizeConstants));
        if (vertexSizeBuffer) {
            mglRenderSetRenderBufferForOwner(
                renderEncoderOwner, vertexSizeBuffer, 0,
                MGL_RENDER_BINDING_STAGE_VERTEX,
                MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX);
            mglBindingRecordLastBoundVertexBuffer(renderer, vertexSizeBuffer, 0,
                                                  MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX);
            MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
        }
    }

    Program *fragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    if (fragmentProgram && fragmentProgram->modules[_FRAGMENT_SHADER].needs_runtime_array_size_buffer)
    {
        uint32_t sizeConstants[31];
        mglSizeConstantsFillFromMapList(&state->fragment_buffer_map_list, sizeConstants);

        void *fragmentSizeBuffer = mglSizeConstantsBufferForStage(
            &areas, MGL_RENDERER_BACKEND_SIZE_CONSTANTS_FRAGMENT,
            sizeConstants, sizeof(sizeConstants));
        if (fragmentSizeBuffer) {
            mglRenderSetRenderBufferForOwner(
                renderEncoderOwner, fragmentSizeBuffer, 0,
                MGL_RENDER_BINDING_STAGE_FRAGMENT,
                MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX);
            mglBindingRecordLastBoundFragmentBuffer(renderer, fragmentSizeBuffer, 0,
                                                    MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX);
            MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
        }
    }

    return true;
}
