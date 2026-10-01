/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_renderer_binding_ops.c — renderer-level forwarders: each one is a single
 * mglRenderBinding* call on the renderer's binding-state owner.
 */

#include "mgl_render_pass_sync_ops.h"
#include "mgl_renderer_binding_ops.h"
#include "mgl_sampled_sampler.h"
#include "mgl_renderer_ports.h"
#include "mgl_stage_encode_drivers.h" /* stage binding drivers (log 131) */
#include "mgl_buffer_map.h"        /* map/update-dirty entries */
#include "mgl_size_constants.h"    /* runtime-array size constants */
#include "mgl_encode_context.h"    /* MGLEncodeContext */
#include "mgl_batch_public.h"       /* mglBatchBindActiveTexturesToMTL */

static void *mglRendererBindingOwner(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    return areas.binding_state_owner ? *areas.binding_state_owner : NULL;
}

void mglRendererBindingInvalidateLastBoundState(void *renderer)
{
    mglRenderBindingInvalidate(mglRendererBindingOwner(renderer));
}

void mglRendererBindingRecordLastBoundVertexBuffer(void *renderer, void *buffer,
                                                   uint64_t offset,
                                                   uint64_t index)
{
    mglRenderBindingRecordVertexBuffer(mglRendererBindingOwner(renderer),
                                       buffer, offset, (uint32_t)index);
}

void mglRendererBindingRecordLastBoundFragmentBuffer(void *renderer,
                                                     void *buffer,
                                                     uint64_t offset,
                                                     uint64_t index)
{
    mglRenderBindingRecordFragmentBuffer(mglRendererBindingOwner(renderer),
                                         buffer, offset, (uint32_t)index);
}

void mglRendererBindingInvalidateLastBoundVertexBufferAtIndex(void *renderer,
                                                              uint64_t index)
{
    mglRenderBindingInvalidateVertexBuffer(mglRendererBindingOwner(renderer),
                                           (uint32_t)index);
}

void mglRendererBindingInvalidateLastBoundFragmentBufferAtIndex(
    void *renderer, uint64_t index)
{
    mglRenderBindingInvalidateFragmentBuffer(mglRendererBindingOwner(renderer),
                                             (uint32_t)index);
}

/* MGL_STATE() from MGLRenderer_Private.h, in C; the argument is the caller's
 * context, exactly as the method's MGL_STATE(glm_ctx) was. */
static GLMState *mglRendererBindingDerivedState(const MGLRendererStateAreas *areas,
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
    mglRendererFillStateAreas(renderer, &areas);
    GLMState *state = mglRendererBindingDerivedState(&areas, glm_ctx);

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
        mglRenderPassRestoreRenderEncoderAfterTextureUpload(
            renderer, "final-active-texture-bind"));
    if (!mglBindTexturesToCurrentRenderEncoder(renderer, &encCtx)) {
        RETURN_FALSE_ON_FAILURE(
            mglRenderPassRestoreRenderEncoderAfterTextureUpload(
                renderer, "final-sampled-texture-bind"));
        RETURN_FALSE_ON_FAILURE(
            mglBindTexturesToCurrentRenderEncoder(renderer, &encCtx));
    }
    return true;
}
