/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_binding_state_ops.c — bodies of the binding-state forwarders that used to
 * sit in MGLRenderer+Binding.m.  Each one is a single mglRenderBinding* call
 * with the binding-state owner from the state areas and, for the owner-aware
 * forms, the current render encoder owner from the command state.
 */

#include "mgl_binding_state_ops.h"
#include "mgl_sampled_sampler.h"
#include "mgl_renderer_ports.h"
#include "mgl_stage_encode_drivers.h" /* stage binding drivers (log 131) */
#include "mgl_buffer_map.h"        /* map/update-dirty entries */
#include "mgl_size_constants.h"    /* runtime-array size constants */
#include "mgl_encode_context.h"    /* MGLEncodeContext */
#include "mgl_batch_issue.h"       /* mglBatchBindActiveTexturesToMTL */

void mglBindingInvalidateLastBoundState(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    mglRenderBindingInvalidate(areas.binding_state_owner
                                   ? *areas.binding_state_owner
                                   : NULL);
}

void mglBindingRecordLastBoundVertexBuffer(void *renderer, void *buffer,
                                           uint64_t offset, uint64_t index)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    mglRenderBindingRecordVertexBuffer(
        areas.binding_state_owner ? *areas.binding_state_owner : NULL, buffer,
        offset, (uint32_t)index);
}

void mglBindingRecordLastBoundFragmentBuffer(void *renderer, void *buffer,
                                             uint64_t offset, uint64_t index)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    mglRenderBindingRecordFragmentBuffer(
        areas.binding_state_owner ? *areas.binding_state_owner : NULL, buffer,
        offset, (uint32_t)index);
}

void mglBindingInvalidateLastBoundVertexBufferAtIndex(void *renderer,
                                                      uint64_t index)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    mglRenderBindingInvalidateVertexBuffer(
        areas.binding_state_owner ? *areas.binding_state_owner : NULL,
        (uint32_t)index);
}

void mglBindingInvalidateLastBoundFragmentBufferAtIndex(void *renderer,
                                                        uint64_t index)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    mglRenderBindingInvalidateFragmentBuffer(
        areas.binding_state_owner ? *areas.binding_state_owner : NULL,
        (uint32_t)index);
}

void mglBindingSetViewportIfNeeded(void *renderer, double origin_x,
                                   double origin_y, double width,
                                   double height, double znear, double zfar)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    void *owner = areas.command ? areas.command->currentRenderEncoderOwner : NULL;
    mglRenderBindingSetViewportForOwner(
        areas.binding_state_owner ? *areas.binding_state_owner : NULL, owner,
        origin_x, origin_y, width, height, znear, zfar);
}

void mglBindingSetScissorRectIfNeeded(void *renderer, int64_t x, int64_t y,
                                      uint64_t width, uint64_t height)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    void *owner = areas.command ? areas.command->currentRenderEncoderOwner : NULL;
    mglRenderBindingSetScissorForOwner(
        areas.binding_state_owner ? *areas.binding_state_owner : NULL, owner,
        x, y, width, height);
}

void mglBindingSetTriangleFillModeIfNeeded(void *renderer, uint32_t mode)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    void *owner = areas.command ? areas.command->currentRenderEncoderOwner : NULL;
    mglRenderBindingSetTriangleFillForOwner(
        areas.binding_state_owner ? *areas.binding_state_owner : NULL, owner,
        mode);
}

/* MGL_STATE() from MGLRenderer_Private.h, in C; the argument is the caller's
 * context, exactly as the method's MGL_STATE(glm_ctx) was. */
static GLMState *mglBindingStateDerivedState(const MGLRendererStateAreas *areas,
                                             GLMContext glm_ctx)
{
    if (areas->core && areas->core->activeState) {
        return areas->core->activeState;
    }
    return glm_ctx ? glm_ctx->active_state : NULL;
}

bool mglRendererSyncResourceBindingsForContext(
    void *renderer, GLMContext glm_ctx, const MGLResourceSyncWork *done)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMState *state = mglBindingStateDerivedState(&areas, glm_ctx);

    if (!done || !done->mappedBuffers) {
        RETURN_FALSE_ON_FAILURE(mglRendererMapBuffersToMTL(renderer));
    }
    if (!done || !done->updatedBaseLists) {
        RETURN_FALSE_ON_FAILURE(
            mglRendererUpdateDirtyBaseBufferList(renderer,
                                                 &state->vertex_buffer_map_list));
        RETURN_FALSE_ON_FAILURE(
            mglRendererUpdateDirtyBaseBufferList(renderer,
                                                 &state->fragment_buffer_map_list));
    }
    MGLEncodeContext encCtx = {
        .render_encoder_owner =
            areas.command ? areas.command->currentRenderEncoderOwner : NULL,
    };
    RETURN_FALSE_ON_FAILURE(
        mglStageEncodeBindVertexBuffers(renderer, &encCtx));
    RETURN_FALSE_ON_FAILURE(
        mglStageEncodeBindFragmentBuffers(renderer, &encCtx));
    RETURN_FALSE_ON_FAILURE(
        mglRendererBindBufferSizeConstantsForRenderEncoder(renderer));
    if (!done || !done->boundActiveTextures) {
        RETURN_FALSE_ON_FAILURE(
            mglBatchBindActiveTexturesToMTL(renderer, areas.ctx));
    }
    RETURN_FALSE_ON_FAILURE(
        mglRendererRestoreRenderEncoderAfterTextureUploadPort(
            renderer, "final-active-texture-bind"));
    if (!mglBindTexturesToCurrentRenderEncoder(renderer, &encCtx)) {
        RETURN_FALSE_ON_FAILURE(
            mglRendererRestoreRenderEncoderAfterTextureUploadPort(
                renderer, "final-sampled-texture-bind"));
        RETURN_FALSE_ON_FAILURE(
            mglBindTexturesToCurrentRenderEncoder(renderer, &encCtx));
    }
    return true;
}
